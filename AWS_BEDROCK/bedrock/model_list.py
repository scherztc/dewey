import boto3

bedrock = boto3.client('bedrock', region_name='us-east-1') 

response = bedrock.list_foundation_models()
print(response)
