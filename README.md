# Running Fargate tasks to extract data from image files using textract

This project helps you extract data from image files containing newpaper clippings. The process contains pre-processing step to split the newspaper page into multiple sections and then extract the text using  Textract

You create a new bucket in S3 and upload image files in Prefix input/.

This triggers a Lambda function which calls Fagate tasks to run a docker container. Container has processing logic to extract text.
2 output files are created for each image file uplaoded that are copied to output/ prefix.

Architecture enables processing a large no of files in parallel. Each time a new image file is uploaded to S3, it calls a Lambda function that invikes a Fargate task - thereby allowing hundreds of images files to be processed in parallel.

Key Services used
- AWS S3 - for uploading image files and also storing the output
- AWS lambda - Triggers the ECS fargate tasks for running Docker container that contains programs for processing image files
- AWS Fargate Task - for procesisng docker container

### Steps to setup the application

<b>Step 1. Create a S3 bucket </b>

 Create a new bucket s3://S3xxxx and following folder structure
  
S3xxxx
  input/ 
      	0001.png
     	0002...N.png
  output/ 
	ocr_plain_text -> plain OCR results as text file. (`.txt`)
	ocr_coords -> OCR results with coordinates (x, y) (`.txt`) and json files



<b>Step 2. Create the Lambda function </b>

Create a new lambda function that will be triggered when a new file is uploaded to S3 bucket created inprevious Step
  - Ensure that Lambda role has access to all resources like  S3, Cloudwatch etc
  - Set the Trigger for Lambda function as S3 created in previous step 
  - Please use the lambda code pasteed below and change the  variables 

import boto3
import json 
import random
import time

client = boto3.client('ecs')
cluster_name = << Your Fargate Cluster >>
task_definition = << Your Task >>
container_name = << Your container >>
subnet_name = << Your subnet name >>
security_group =  <<Your secuirty group>>
  
  

def lambda_handler(event, context):
    try:
        
   
        lambda_message = event['Records'][0]
        bucket = lambda_message['s3']['bucket']['name']
        key = lambda_message['s3']['object']['key']
        print('KEY:'+key)
        
        img_file =  key.split('/')[-1]
        img_file2 = img_file.split('.')[0]+'.txt'
        img_file3 = img_file.split('.')[0]+'.json'
        
        
        response = client.run_task(
            cluster=cluster_name,
            launchType = 'FARGATE',
            taskDefinition=task_definition,
            count = 1,
            platformVersion='LATEST',
            overrides={
                'containerOverrides':[{
                    'name': container_name,
                    'command':[img_file,img_file2,img_file3]
                }],
            },
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [
                        subnet_name
                    ],
                    'securityGroups': [
                        security_group,
                    ],
                    'assignPublicIp': 'ENABLED'
                }
            })

        print(response)

        return {
            'statusCode': 200,
            'body': "OK"
        }
    except Exception as e:
        print(e)

        return {
            'statusCode': 500,
            'body': str(e)
        }    
	
	
