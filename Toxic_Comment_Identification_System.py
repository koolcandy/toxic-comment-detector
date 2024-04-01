import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from utils import train_model
from utils import check_words
from utils import get_result_by_local_model
from utils import get_result_by_gemini
from utils import config
import logging
import urllib.parse
# encoding:utf-8

def get_result(words):
    # log_file = os.path.join(config.work_dir, 'log.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO)
    # # Add console logging

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)

    # if not logging.getLogger('').hasHandlers():
    #     print('add console')
    #     logging.getLogger('').addHandler(console)
    
    if check_words.main(words):
        print('Local bad word matched')
        result = {
            'toxic': 1,
            'severe_toxic': 0,
            'obscene': 0,
            'threat': 0,
            'insult': 0,
            'identity_hate': 0
        }
        return result
    else:
        print('Local bad words not matched, start to use local model to predict...')
        result = get_result_by_local_model.main(words)
        print('Local model result: %s' % result)
        toxic_score = result['toxic']
        if toxic_score > 0.3:
            return result
        else:
            print('Local model not matched, start to use Gemini model to predict...')
            gemini_result = get_result_by_gemini.main(words)
            print('Gemini model result: %s' % gemini_result)
            return gemini_result

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = urllib.parse.parse_qs(post_data.decode('utf-8'))
        text = post_data.get('input', [''])[0]
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