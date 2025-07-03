import pandas as pd
import numpy as np
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, ArrayType, StringType
from pyspark.sql.functions import col, pandas_udf, explode, window, approx_count_distinct
from pyspark.sql.protobuf.functions import from_protobuf
from pyspark.sql.pandas.functions import PandasUDFType
from typing import Iterator

print(f"================> DRIVER PROCESS STARTED. PID: {os.getpid()} <================")

# --- Khởi tạo SparkSession ---
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("kafka-yolo-protobuf-consumer") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "8")

# --- Đọc dữ liệu từ Kafka Stream ---
topic = "count-vehicle"
kafkaDf = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", topic) \
    .option("startingOffsets", "latest") \
    .load()

# --- Bước 3: Giải mã Protobuf ---
PROTO_DESCRIPTOR_FILE = "frame-message.desc"
decodedDf = kafkaDf.withColumn(
    "payload",
    from_protobuf(
        data=col("value"),
        messageName="VideoPayload",
        descFilePath=PROTO_DESCRIPTOR_FILE
    )
).select(
    "timestamp",
    col("payload.camera_id").alias("camera_id"),
    col("payload.frame").alias("frame_bytes")
)

# --- Định nghĩa SCHEMA, PANDAS UDF ---
box_schema = StructType([
    StructField("xmin", FloatType(), nullable=False), StructField("ymin", FloatType(), nullable=False),
    StructField("xmax", FloatType(), nullable=False), StructField("ymax", FloatType(), nullable=False)
])
detection_schema = StructType([
    StructField("box", box_schema, nullable=False), StructField("confidence", FloatType(), nullable=False),
    StructField("class_id", IntegerType(), nullable=False), StructField("class_name", StringType(), nullable=False),
    StructField("track_id", IntegerType(), nullable=True)
])
YOLO_OUTPUT_SCHEMA = ArrayType(detection_schema)

class YoloTracker:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO("custom-train-yolov8n/weights/best.pt")
        self.class_names = self.model.names

    def track_batch(self, series: pd.Series) -> pd.Series:
        import io; from PIL import Image
        pil_images = [Image.open(io.BytesIO(b)) if b is not None else None for b in series]
        valid_images = [img for img in pil_images if img is not None]
        if not valid_images: return pd.Series([[] for _ in pil_images])

        numpy_images = [np.array(img) for img in valid_images]
        result_batch = self.model.track(numpy_images, persist=True, verbose=False)
        result_iterator = iter(result_batch)

        final_results = []
        for img in pil_images:
            if img is None:
                final_results.append([])
                continue
            
            results = next(result_iterator)
            detections = []
            for box in results.boxes:
                xyxy = box.xyxy[0].tolist(); conf = box.conf[0].item()
                cls_id = int(box.cls[0].item()); track_id = int(box.id[0].item()) if box.id is not None else None
                detections.append(((xyxy[0], xyxy[1], xyxy[2], xyxy[3]), conf, cls_id, self.class_names[cls_id], track_id))
            final_results.append(detections)
        return pd.Series(final_results)

@pandas_udf(YOLO_OUTPUT_SCHEMA, PandasUDFType.SCALAR_ITER)
def yolo_iterator_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    tracker = YoloTracker()
    for series in iterator:
        yield tracker.track_batch(series)

predictionsDf = decodedDf.withColumn("tracked_objects", yolo_iterator_udf(col("frame_bytes")))
explodedDf = predictionsDf.select(col("timestamp"), col("camera_id"), explode(col("tracked_objects")).alias("single_object"))

flatDf = explodedDf.select(
    "timestamp",
    "camera_id",
    col("single_object.class_name").alias("class_name"),
    col("single_object.track_id").alias("track_id")
)

# --- watermark cho phép  data có thể đến trễ ---
delayed_time = 2
flatDf = flatDf.withWatermark("timestamp", f"{delayed_time} seconds")

# --- GroupBy theo cửa sổ thời gian, camera_id, class_name ---
# --- Thực hiện đếm các track_id khác nhau của từng class_name ---
window_time = 10
windowedCountsDf = flatDf \
    .groupBy(
        window(col("timestamp"), f"{window_time} seconds"),
        col("camera_id"),
        col("class_name")
    ) \
    .agg(
        approx_count_distinct("track_id").alias("vehicle_count")
    )

# --- Format lại các cột để hiển thị cho gọn gàng ---
finalDf = windowedCountsDf.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    "camera_id",
    "class_name",
    "vehicle_count"
)

# --- Ghi kết quả đã tổng hợp ra file CSV ---
query = finalDf.repartition(1).writeStream \
    .trigger(processingTime="5 seconds") \
    .outputMode("append") \
    .format("csv") \
    .option("path", "results")  \
    .option("header", "true") \
    .option("checkpointLocation", "spark_checkpoint") \
    .start()

query.awaitTermination()