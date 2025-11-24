from confluent_kafka import Consumer, KafkaError
import socket
import json

BROKER = 'localhost:9092'
GROUP_ID = 'analytics'
TOPIC = 'youtube_topic'


def create_kafka_consumer(broker, group_id, topic):
    conf = {
        'bootstrap.servers': broker,
        'group.id': group_id,
        'auto.offset.reset': 'latest',
        'client.id': socket.gethostname()
    }
    consumer = Consumer(conf)
    consumer.subscribe([topic])
    return consumer


def display_kafka_data():
    consumer = create_kafka_consumer(BROKER, GROUP_ID, TOPIC)
    print("Listening to Kafka topic:", TOPIC)

    video_results = []
    max_messages = 5

    while len(video_results) < max_messages:
        msg = consumer.poll(timeout=3.0)
        if msg is None:
            break
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                break
            else:
                print(msg.error())
                break
        try:
            video_data = json.loads(msg.value().decode('utf-8'))
        except Exception:
            continue

        if "total_comment_likes" in video_data:  # don't include some not needed message
            video_results.append(video_data)
            print(f"The title is {video_data['title']}")

    consumer.close()

    if not video_results:
        print("Didn't get anything.")
        return

    most_popular_video = max(video_results, key=lambda v: v['total_comment_likes'])
    print(f"The most popular video is: {most_popular_video['title']}")
    print(f"Total number of comment likes: {most_popular_video['total_comment_likes']}")


if __name__ == "__main__":
    display_kafka_data()
