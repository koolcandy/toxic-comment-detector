import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from cleantext import clean

import utils.train_model as train_model
import utils.check_words as check_words
import utils.get_result_by_local_model as get_result_by_local_model
import utils.get_result_by_gemini as get_result_by_gemini
# encoding:utf-8

import urllib.parse
import google.generativeai as genai

work_dir = os.path.dirname(os.path.abspath(__file__))

def get_result(words):
    if check_words.main(words):
        print('Local bad words not matched')
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
        print('Local model result:', result)
        toxic_score = result['toxic']
        if toxic_score > 0.3:
            return result
        else:
            print('Local model not matched, start to use Gemini model to predict...')
            return get_result_by_gemini.main(words)

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
    if not os.path.exists(os.path.join(work_dir, 'model')):
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