import pika
import json
import sys
import os

# Import your actual prediction function from your existing model file
# (Adjust 'model' to whatever your file is named if it's different)
from model import predict 

def callback(ch, method, properties, body):
    """This function runs automatically every time a new message arrives."""
    try:
        # 1. Parse the incoming message (The REST API will send this JSON)
        data = json.loads(body)
        image_id = data.get("image_id")
        image_path = data.get("image_path")
        
        print(f" [▼] Received task for Image ID: {image_id}")
        print(" [⧗] Starting deep learning inference...")

        # 2. Run the actual AI Analysis (F-05)
        # This calls your PyTorch code!
        results = predict(image_path) 
        
        # 3. Format the result payload
        # Ensure 'stone_coverage' is calculated as per your F-05 specs
        result_payload = {
            "status": "RESULT_READY",
            "image_id": image_id,
            "diagnosis": results.get("diagnosis", "Unknown"),
            "confidence": results.get("confidence", 0.0),
            "stone_coverage": results.get("stone_coverage", 0.0)
        }

        # 4. Publish to notifications_queue (F-09)
        # This tells the REST API that the work is done
        ch.basic_publish(
            exchange='',
            routing_key='notifications_queue',
            body=json.dumps(result_payload)
        )
        
        # 5. Acknowledge the message so RabbitMQ removes it from the queue
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f" [▲] Finished and notified REST API for Image ID: {image_id}\n")

    except Exception as e:
        print(f" [!] Error processing image: {e}")
        # If it fails, we reject it so it doesn't get stuck forever
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

def main():
    print(" [*] Connecting to RabbitMQ broker...")
    
    # CONNECTION NOTE: 
    # Because we will run this inside Docker, the host is the container name 'rabbitmq'.
    # (If you ever test this directly in VS Code outside of Docker, change this to 'localhost')
    credentials = pika.PlainCredentials('admin', 'admin123')
    connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='rabbitmq', credentials=credentials)
      )
    channel = connection.channel()

    # Ensure both queues exist before we try to use them
    channel.queue_declare(queue='analysis_queue', durable=True)
    channel.queue_declare(queue='notifications_queue', durable=True)

    # Don't give more than 1 message to a worker at a time (prevents memory crashes)
    channel.basic_qos(prefetch_count=1) 
    
    # Tell RabbitMQ to trigger the 'callback' function when a message arrives
    channel.basic_consume(queue='analysis_queue', on_message_callback=callback)

    print(' [*] Worker is alive and listening to "analysis_queue". To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)