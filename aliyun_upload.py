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
from dotenv import load_dotenv
from pdf_generate import generate_pdf

load_dotenv()

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
def upload_to_oss(user_id):
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


    ###################################################################################################################
    # Example face analysis result (your provided data)
    result = {
        "Face_shape": {"三庭比例": "1 : 1.21 : 1.05", "三线比例": "0.95 : 1 : 0.86", "下巴形状": "钝弧（圆形下巴）", "五眼比例": "0.62 : 1 : 1.15 : 1.02 : 0.6",
                       "脸型判断结果": "菱形脸", "脸型曲直": "偏直", "脸部风格": "御姐成熟脸", "脸长和脸宽的 比例": "1.35"},
        "Face_shape_info": {"三庭比例": "1 : 1.21 : 1.05", "三线比例": "0.95 : 1 : 0.86", "下巴形状": "钝弧（圆 形下巴）", "五眼比例": "0.62 : 1 : 1.15 : 1.02 : 0.6",
                            "脸型判断结果": "菱形脸", "脸型曲直": "偏直", "脸部风格": "御姐成熟脸", "脸长和脸宽的比例": "1.35"},
        "Lips_detailed_info": {"上下唇比例": 1.546, "上唇": "偏厚偏曲", "下唇": "偏薄偏直", "唇形": "性感M唇", "唇部数据": "圆唇, 偏薄, M型唇峰", "曲直结果": "偏曲"},
        "eye_detailed_info": {"右眼曲直": "偏直", "右眼眼长和眼高的比例": "2.27", "右眼类型": "细长眼", "左眼曲直": "偏直", "左眼眼长和眼高的比例": "2.75",
                              "左 眼类型": "柳叶眼", "眼型综合曲直": "偏直"},
        "eye_shape": {"右眼内眼角角度": "57.48°", "右眼曲直": "偏直", "右眼特征": "细长", "右眼类型": "细长眼", "右眼长高比例": "2.27",
                      "左眼内眼角角度": "41.19°", "左眼曲直": "偏直", "左眼特征": "细长", "左眼类型": "柳叶眼", "左眼长高比例": "2.75", "眼型曲直综合": "偏直"},
        "lip_shape": {"上下唇比例": 1.546, "上唇": "偏厚偏曲", "下唇": "偏薄偏直", "唇形": "性感M唇", "唇部数据": "圆唇, 偏薄, M型唇峰",
                      "嘴角": "嘴角平坦, 右嘴角 倾斜度:0.02°, 左嘴角倾斜度:0.02°", "曲直结果": "偏曲"},
        "nose_detailed_info": {"右脸山根": "山根靠上, 偏直", "右鼻翼曲率": "0.666", "山根曲直": "偏直", "左脸山根": "山根靠上, 偏直",
                               "左鼻翼曲率": "0.333", "鼻型综合曲直": "偏直", "鼻孔比例": "38.28%", "鼻翼宽窄判断": "宽鼻翼,偏曲", "鼻翼曲直判断": "偏曲"},
        "nose_shape": {"右脸山根": "山根靠上, 偏直", "右鼻翼曲率": "0.666", "山根曲直": "偏直", "左脸山根": "山根靠上, 偏直",
                       "左鼻翼曲率": "0.333", "鼻型综合曲直": "偏直", "鼻孔曲直": "偏曲", "鼻孔比例": "38.28%", "鼻翼宽窄": "宽鼻翼,偏曲", "鼻翼曲线综合判断": "偏曲"}
    }


    pdf_buffer = generate_pdf(result)

    # Save the PDF to a local file
    with open("face_analysis_report.pdf", "wb") as f:
        f.write(pdf_buffer.getvalue())
    ###################################################################################################################


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
