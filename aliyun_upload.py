import os
import json
import base64
import hmac
import datetime
import time
import hashlib
import requests
from alibabacloud_tea_openapi.models import Config
from alibabacloud_sts20150401.client import Client as Sts20150401Client
from alibabacloud_sts20150401 import models as sts_20150401_models

# Environment variable configuration
access_key_id = os.environ.get('OSS_ACCESS_KEY_ID')
access_key_secret = os.environ.get('OSS_ACCESS_KEY_SECRET')
role_arn_for_oss_upload = os.environ.get('OSS_STS_ROLE_ARN')

role_session_name = 'role_session_name'

# OSS settings
bucket = 'seasons-upload'
region_id = 'cn-beijing'
host = f'http://{bucket}.oss-cn-beijing.aliyuncs.com'
upload_dir = 'dir'
local_file_path = 'requirements.txt'

# Function to calculate HMAC-SHA256
def hmacsha256(key, data):
    try:
        mac = hmac.new(key, data.encode(), hashlib.sha256)
        return mac.digest()
    except Exception as e:
        raise RuntimeError(f"Failed to calculate HMAC-SHA256 due to {e}")

# Function to handle the file upload to OSS
def upload_to_oss():
    # Initialize STS client and get temporary credentials
    config = Config(
        region_id=region_id,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret
    )
    sts_client = Sts20150401Client(config=config)
    assume_role_request = sts_20150401_models.AssumeRoleRequest(
        role_arn=role_arn_for_oss_upload,
        role_session_name=role_session_name
    )
    response = sts_client.assume_role(assume_role_request)
    token_data = response.body.credentials.to_map()

    temp_access_key_id = token_data['AccessKeyId']
    temp_access_key_secret = token_data['AccessKeySecret']
    security_token = token_data['SecurityToken']

    # Generate date and expiration time
    now = int(time.time())
    dt_obj = datetime.datetime.utcfromtimestamp(now)
    dt_obj_3h = dt_obj + datetime.timedelta(hours=3)

    dt_obj_1 = dt_obj.strftime('%Y%m%dT%H%M%S') + 'Z'
    dt_obj_2 = dt_obj.strftime('%Y%m%d')
    expiration_time = dt_obj_3h.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    # Construct Policy
    policy = {
        "expiration": expiration_time,
        "conditions": [
            ["eq", "$success_action_status", "200"],
            {"x-oss-signature-version": "OSS4-HMAC-SHA256"},
            {"x-oss-credential": f"{temp_access_key_id}/{dt_obj_2}/cn-beijing/oss/aliyun_v4_request"},
            {"x-oss-security-token": security_token},
            {"x-oss-date": dt_obj_1},
            ["starts-with", "$key", upload_dir]
        ]
    }
    policy_str = json.dumps(policy).strip()
    base64_policy = base64.b64encode(policy_str.encode()).decode()

    # Generate signing key
    date_key = hmacsha256(("aliyun_v4" + temp_access_key_secret).encode(), dt_obj_2)
    date_region_key = hmacsha256(date_key, "cn-beijing")
    date_region_service_key = hmacsha256(date_region_key, "oss")
    signing_key = hmacsha256(date_region_service_key, "aliyun_v4_request")

    # Generate signature
    result = hmacsha256(signing_key, base64_policy)
    signature = result.hex()

    # Prepare upload parameters
    key = f"{upload_dir}/requirements.txt"
    fields = {
        'key': key,
        'policy': base64_policy,
        'x-oss-signature-version': "OSS4-HMAC-SHA256",
        'x-oss-credential': f"{temp_access_key_id}/{dt_obj_2}/cn-beijing/oss/aliyun_v4_request",
        'x-oss-date': dt_obj_1,
        'x-oss-security-token': security_token,
        'x-oss-signature': signature,
        'success_action_status': '200'
    }

    # Debugging: log fields
    print("Form fields:", fields)

    # Perform the file upload
    with open(local_file_path, 'rb') as file:
        files = {'file': (local_file_path, file)}
        response = requests.post(host, data=fields, files=files)

    if response.status_code == 200:
        return {'message': 'File uploaded successfully', 'file_url': f"{host}/{key}"}
    else:
        return {'message': 'File upload failed', 'status_code': response.status_code, 'response': response.text}
