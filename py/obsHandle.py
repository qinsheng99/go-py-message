import os
import traceback
from obs import ObsClient

obs_sk = os.getenv("OBS_SK")
obs_ak = os.getenv("OBS_AK")
obs_bucketname = os.getenv("OBS_BUCKETNAME")
obs_endpoint = os.getenv("OBS_ENDPOINT")




class OBSHandler:
    def __init__(self):
        self.access_key = obs_ak
        self.secret_key = obs_sk
        self.bucket_name = obs_bucketname
        self.endpoint = obs_endpoint
        self.server = "https://" + obs_endpoint
        self.obsClient = self.init_obs()
        self.maxkeys = 5000  # 查询的对象最大个数

    # 初始化obs
    def init_obs(self):
        obsClient = ObsClient(
            access_key_id=self.access_key,
            secret_access_key=self.secret_key,
            server=self.server
        )
        return obsClient

    def close_obs(self):
        self.obsClient.close()

    def readFile(self, path):
        """
        二进制读取配置文件
        :param path:
        :return:
        """
        try:
            resp = self.obsClient.getObject(self.bucket_name, path, loadStreamInMemory=True)
            if resp.status < 300:
                # 获取对象内容
                return {
                    "status": 200,
                    "msg": "获取配置文件成功",
                    "content": bytes.decode(resp.body.buffer, "utf-8"),
                    "size": resp.body.size
                }
            else:
                return {
                    "status": -1,
                    "msg": "获取失败，失败码: %s\t 失败消息: %s" % (resp.errorCode, resp.errorMessage),
                    "content": "",
                    "size": 0
                }
        except:
            print(traceback.format_exc())

    def downloadFile(self, source_dir, dest_dir):
        response_msg = {'status': 200, 'msg': '单个对象下载成功'}
        res = self.obsClient.getObject(self.bucket_name, source_dir, dest_dir)
        if res.status >= 300:
            response_msg["status"] = -1
            response_msg["msg"] = "单个对象下载失败"
        return response_msg


if __name__ == "__main__":
    obs = OBSHandler()
    # 关闭obsClient
    obs.close_obs()
