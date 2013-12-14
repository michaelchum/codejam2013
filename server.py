# Use with: curl -i -F name=test -F filedata=@/etc/passwd http://127.0.0.1:8080/upload

from bottle import route, request, static_file, run
import os
PROJECT_PATH = os.getcwd()

@route('/upload', method='POST')
def upload():
    #print bottle.request.files['filedata']
    file_path = "~/codejam"
    upload = request.files.get('upload')
    file_path = "{path}/{file}".format(path=PROJECT_PATH, file="test.txt")
    with open(file_path, 'w') as open_file:
		open_file.write(upload.file.read().encode('ISO-8859-1'))



run()