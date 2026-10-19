import sys
from kafka import KafkaAdminClient
from kafka.admin import NewTopic

BOOTSTRAP_SERVERS = "localhost:9092"
TEST_TOPIC = "telemetry.tactical.raw"

def verify_kafka_cluster():
    print(f"Connecting to Kafka cluster at {BOOTSTRAP_SERVERS}...")
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            client_id="health-check-admin"
        )

        existing_topics = admin_client.list_topics()
        print("Connection established successfully.")
        print(f"Existing topics: {existing_topics}")

        if TEST_TOPIC not in existing_topics:
            print(f"Creating project topic: '{TEST_TOPIC}'...")
            new_topic = NewTopic(name=TEST_TOPIC, num_partitions=1, replication_factor=1)
            admin_client.create_topics(new_topics=[new_topic], validate_only=False)
            print(f"Topic '{TEST_TOPIC}' created successfully.")
        else:
            print(f"Topic '{TEST_TOPIC}' already exists.")

        admin_client.close()
        print("\nVerification successful: Kafka environment is operational.")
    except Exception as e:
        print(f"\nConnection failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    verify_kafka_cluster()