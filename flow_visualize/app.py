import os


from jina import Flow
from docarray import Document, DocumentArray
# from jina.logging.predefined import default_logger as logger
# from jina.types.request import Request

def config():
    print("Config Start")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(cur_dir, 'models_data_test')
    workspace_dir = os.path.join(cur_dir, 'workspace')
    os.environ['JINA_PORT'] = '45680'  # the port for accessing the RESTful service, i.e. http://localhost:45680/docs
    os.environ['MODEL_MOUNT'] =  f'{model_dir}:/workspace/models_data_test' # the directory to store the indexed data
    print("Config Success")

def main():
    config()
    
    flow = Flow().load_config('index-flow-visual.yml')
    flow.plot('index-flow.svg')
    
    flow2 = Flow().load_config('search-flow-visual.yml')
    flow2.plot('search-flow.svg')
    

if __name__ == '__main__':
    main()
    
        