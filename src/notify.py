import boto3
import json
from dotenv import load_dotenv

load_dotenv()

def notify(tenant, id, success, sns_topic_arn, data={}):
    try:
        message = {
            'id': id,
            'tenantId': tenant,
            'type': 'video.container.processed',
            'success': success,
            'data': json.dumps(data)
        }

        sns = boto3.client('sns')

        response = sns.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps(message),
            Subject=f"Video Job {id} - {'Success' if success else 'Failed'}"
        )

        print(f"SNS notification sent successfully: {response['MessageId']}")
        print(f"Message: {json.dumps(message, indent=2)}")

    except Exception as e:
        print(f"Error sending SNS notification: {str(e)}")

