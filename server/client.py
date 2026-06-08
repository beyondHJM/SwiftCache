import requests

def send_request(
    method: str,
    url: str,
    query_params: dict = None,
    body: dict = None,
    headers: dict = None
):
    """
    发送 HTTP 请求。

    :param method: HTTP 方法 ('GET', 'POST', 'PUT', 'DELETE' 等)
    :param url: 请求的 URL
    :param query_params: URL 查询参数 (字典)，会自动拼到 URL 后
    :param body: 请求 Body 数据 (字典)，会自动转成 JSON
    :param headers: 请求头 (字典)
    :return: 响应对象 (requests.Response)
    """
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            params=query_params,  # Query 参数
            json=body,            # JSON Body
            headers=headers        # 请求头
        )
        return response
    except Exception as e:
        print(f"发送请求时发生错误: {e}")
        return None


if __name__ == "__main__":
    # 示例调用
    url = "http://127.0.0.1:8000/generate"

    query = {"name": "Tom", "age": 18}   # Query 参数
    body = {"prompt": "Hello World", "max_tokens": 100}  # JSON Body
    headers = {"Authorization": "Bearer your_token_here"}  # 请求头

    resp = send_request("POST", url, query_params=query, body=body, headers=headers)

    if resp is not None:
        print("状态码:", resp.status_code)
        print("响应内容:", resp.text)
