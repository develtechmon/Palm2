## Getting Started

Visit `https://makersuite.google.com/app/apikey` to obtain your API key
```
Key for Palm2 : AIzaSyCqeMc0E7WcuMmRHotLGfuIDdS-TT5zTso
```

1. Install following library
```
!pip install -q llama-index
!pip install pypdf
!pip install google-generativeai
!pip install transformers
!pip install llama-cpp-python
```

2. Import all the required libraries
```
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.palm import PaLM
from IPython.display import Markdown, display
from llama_index import ServiceContext
from llama_index import StorageContext, load_index_from_storage
import os
```

3. Load the keys
```
os.environ['GOOGLE_API_KEY'] = "AIzaSyCqeMc0E7WcuMmRHotLGfuIDdS-TT5zTso"

llm = PaLM()
```