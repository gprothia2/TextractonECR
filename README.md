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

<b>Step 1 -  Create a S3 bucket </b>

 Create a new bucket s3://S3xxxx and following folder structure
  
	S3xxxx
		├── input/ 
		│     └──  0001.png
		│     └──  0002...N.png
		├── output/ 
		│     ├── ocr_plain_text -> plain OCR results as text file. (`.txt`)
		│     └── ocr_coords -> OCR results with coordinates (x, y) (`.txt`) and json files

	
<b> Step 2 - Configure settings for the program </b>

Configure theS3 bucket and other settings for the program

- Go to directory main/EulerEye-master3/src/constants.py and change name of the S3 bucket and folder to new bucket/folder created in previous step
     
		S3_BUCKET_NAME = 'S3xxxx'
		S3_RAW_IMAGE_FOLDER = 'S3xxxx/input/'
		TEXTRACT_LIMIT_SIZE = 10485760
		FCN_MODEL = 'last_20.pth.tar'

- Go to directory main/EulerEye-master3/runscript and change name of the bucket  in Line 3 

		 echo $1
		 export AWS_DEFAULT_REGION=us-east-1
		 bucket_name='xxx'

		 python3 /home/ec2-user/EulerEye-master3/src/pipeline.py --img $1 --root /home/ec2-user/EulerEye-master3 --coords 2
		 aws s3 cp /home/ec2-user/EulerEye-master3/output/ocr_plain_text/$2 s3://$bucket_name/output/ocr_plain_text/$2
		 aws s3 cp /home/ec2-user/EulerEye-master3/output/ocr_coords/$2 s3://$bucket_name/output/ocr_coords/$2
		 aws s3 cp /home/ec2-user/EulerEye-master3/output/ocr_coords/$3 s3://$bucket_name/output/ocr_coords/$3
	 
<b> Step 3 -  Build the docker container and load to ECR </b>

- Go to file main/docker_utils/build  - change name of the application

- Go to file main/docker_utils/load_ecr  - chanage the name of AWS account and Image

		Execute "sh build" - this will build the docer image locally

		Execute "Sh load_ecr" - this will load the dociker image to AWS ECR
	


<b> Step 4 -  Create the Fargate tasks and ECS </b>

 - Go to AWS Console and register task that points to container uploaded in previous step

 - Create a new Cluster and configure Service to point to the task



<b>Step 5. Create the Lambda function </b>

Create a new lambda function that will be triggered when a new file is uploaded to S3 bucket created inprevious Step
   - Ensure that Lambda role has access to all resources like  S3, Cloudwatch etc
   - Set the Trigger for Lambda function as S3 bucket and folder input created in previous step 
   - Please use the lambda code pasted below and change the  variables 

	import boto3
	import json 
	import random
	import time

	cluster_name = << Your Fargate Cluster >>
	task_definition = << Your Task >>
	container_name = << Your container >>
	subnet_name = << Your subnet name >>
	security_group =  <<Your secuirty group>>

	client = boto3.client('ecs')

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


<b> Step 6. Load files to S3 input folder and validate the output </b>

  - Load around 10 files. Each file will invoke the Lambda function that will run the Fargate tasks and extract the text.
  - Extracted file will be loaded to the output folder
  - Validate the outputs and load and process more files

