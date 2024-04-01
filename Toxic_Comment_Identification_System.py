import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from utils import train_model
from utils import get_result_by_local_model
from utils import config
import urllib.parse
# encoding:utf-8

def get_result(words):

    print('Local bad words not matched, start to use local model to predict...')
    result = get_result_by_local_model.main(words)
    print('Local model result: %s' % result)

    return result

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path).query
        query_components = urllib.parse.parse_qs(query)
        text = query_components.get('input', [''])[0]
        result = get_result(text)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        # 添加跨域支持的头部信息
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    #http服务器

def main():
    if not os.path.exists(os.path.join(config.work_dir, 'model')):
        print('Local models not founded, start raining model...')
        train_model.main()
        print('Local models trained successfully.')
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Running server...')
    httpd.serve_forever()

if __name__ == '__main__':
    main()
#驱动代码