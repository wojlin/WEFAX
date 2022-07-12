import pytest
import pytest_check as check
import logging
import os
from os import listdir
from os.path import isfile, join
import importlib

from wefax import Demodulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('my-even-logger')


def check_file_presence(path):
    if os.path.isfile(path):
        logger.info(f'directory "{path}" exist')
        return True
    else:
        logger.error(f'directory "{path}" does not exist')
        return False


def check_directory_presence(path):
    if os.path.isdir(path):
        logger.info(f'directory "{path}" exist')
        return True
    else:
        logger.error(f'directory "{path}" does not exist')
        return False


def test_project_directory_structure():
    directories = ['static', 'templates', 'test_files', 'static/js', 'dummy', 'static/css', 'static/gallery',
                   'static/images',
                   'static/temp']
    for directory in directories:
        check.equal(check_directory_presence(directory), True)


def test_project_files_structure():
    py_files = ['main.py', 'progress_bar.py', 'wefax.py', 'wefax_live.py']
    html_files = ['templates/base.html', 'templates/index.html', 'templates/file_converter.html',
                  'templates/live_converter.html', 'templates/gallery.html']
    css_files = ['static/css/file_converter.css', 'static/css/live_converter.css', 'static/css/index.css',
                 'static/css/gallery.css', 'static/css/style.css']
    js_files = ['static/js/convert_file.js', 'static/js/progress_bar.js', 'static/js/upload.js', 'static/js/socketio']

    for file in py_files:
        check.equal(check_file_presence(file), True)

    for file in html_files:
        check.equal(check_file_presence(file), True)

    for file in css_files:
        check.equal(check_file_presence(file), True)

    for file in js_files:
        check.equal(check_file_presence(file), True)


def test_wefax_filetype_input():
    test_files_path = 'test_files/'
    onlyfiles = [f for f in listdir(test_files_path) if isfile(join(test_files_path, f))]
    for file in onlyfiles:
        file_ext = str(file).split('.')[-1]
        try:
            demodulator = Demodulator(test_files_path + file, lines_per_minute=120, tcp_stream=True, quiet=True)
            logger.info(f'demodulator worked for file: {test_files_path}{file}')
        except Exception as e:
            if str(e) == "INVALID FILETYPE: only .wav files are supported at this moment" and file_ext != 'wav':
                logger.info(f'invalid filetype detected for invalid file: {test_files_path}{file}')
            elif str(e) == "INVALID FILETYPE: only .wav files are supported at this moment" and file_ext == 'wav':
                logger.error(f'invalid filetype detected for a valid file: : {test_files_path}{file}')


def test_imported_packages():
    modules = ['flask', 'flask_socketio', 'werkzeug', 'numpy', 'scipy', 'PIL', 'matplotlib', 'pyaudio', 'wave']
    for module in modules:
        try:
            importlib.import_module(module)
            logger.info(f'{module} imported!')
        except ImportError:
            logger.error(f'{module} not imported!')


if __name__ == '__main__':
    pytest.main(args=[os.path.abspath(__file__), '--html=report.html', '-v', '--self-contained-html', '--color=yes',
                      '--show-capture=log', '--capture=tee-sys'])
