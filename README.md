# Real-time Traffic Video Analysis with Spark and Kafka

## Introduction

This is a Big Data project for real-time video analysis to count traffic vehicles. The project utilizes Apache Kafka for streaming video data and Apache Spark for processing and analyzing this data. An object detection model is employed to identify and count various vehicles such as cars, motorcycles, buses, and trucks.

### Key Features:

* **Real-time Video Processing:** Analyze live video streams from sources like traffic cameras.
* **Distributed System:** Built on the Kafka and Spark platform, allowing for large-scale data processing and scalability.

## System Architecture

The system is designed based on the producer-consumer model:

1.  **Producer:** A Python script (using OpenCV) reads video from a source and splits it into frames. Each frame is then **serialized** using **Protocol Buffers** before being sent to a Kafka topic.
2.  **Kafka:** Acts as a message broker, receiving serialized messages from the producer and allowing consumers to read this data in parallel.
3.  **Consumer (Spark Streaming):** Reads data from Kafka, **deserializes** the messages using **Protocol Buffers** to retrieve the frame data. It then applies the **YOLO (You Only Look Once)** object detection model to identify and count vehicles within each frame. The final results are aggregated and stored.

## Technologies Used

* **Apache Kafka:** Used to build the streaming data pipeline.
* **Apache Spark:** Specifically Spark Streaming for real-time data processing.
* **Protocol Buffers:** Used for serializing and deserializing data to optimize performance.
* **Python:** The primary language used for development.
* **OpenCV:** A library for image and video processing.
* **YOLO (You Only Look Once):** A deep learning model used for object detection.
