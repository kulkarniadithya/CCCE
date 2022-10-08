<html>
<body>
<h2>CCCE: An Unsupervised Approach for Emotion Cause and Category Extraction</h2>

The current SOTA performance is achieved by supervised methods that require large-scale human-annotated training data. In this work, we aim to address this challenge and conduct ECE and ECAE tasks in an unsupervised manner. To our knowledge, this is the first work that tackles both ECE and ECAE tasks in an unsupervised manner.

<h3>Requirements</h3>
``Run requirements.txt to download the required dependencies.``

<h3> Dataset </h3>

The performance of the proposed CCCE is evaluated on two widely used datasets - Chinese [gui et. al, 2016] (2,105 documents) collected from SINA city news (http://news.sina.com.cn/society/), and English [gao et. al, 2017] (2145 documents) collected from English novels. The sample dataset is provided in the folder ``sample_data``. [We cannot share the entire data due to data sharing restrictions]

<h3> Pre-processing </h3>
Data pre-processing includes four steps:
<ul>
<li>Process original data: The original file is processed and stored in a dictionary.

```Run code/pre-processing/process_original_data.py to perform this step.```
</li>
<li>Parser: In this step, the CoreNLP dependency parser is executed
to get dependency relations. 
Download the CORENLP jar files from https://stanfordnlp.github.io/CoreNLP/download.html.
Place stanford-corenlp-4.0.0.jar, stanford-corenlp-4.0.0-models-chinese and stanford-corenlp-4.0.0-models.jar in dependency_parser folder and run the following command from the folder.

```English: java -Xmx8g -XX:-UseGCOverheadLimit -XX:MaxPermSize=1024m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9015  -port 9015 -timeout 1500000```
```Chinese: java -Xmx8g -XX:-UseGCOverheadLimit -XX:MaxPermSize=1024m -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9005  -port 9005 -timeout 1500000```

Once the process is running on port 9015/9005, 

```Run code/pre-processing/parser.py to generate weak labels.```

</li>
<li>convert_to_index: Since tokens can be repeated in the document, we use indexes to obtain weak labels.

```Run code/pre-processing/convert_to_index.py to perform this step.```

</li>

<li>
Weak label generator: In this step, we generate weak labels for ECE and ECAE tasks.

```ECE task: Run code/pre-processing/get_pseudo_labels.py```
```ECAE task: Run code/pre-processing/category_pseudo_labels.py```

</li>

</ul>

<h3>Model</h3>
<ul>
<li>

```Run code/model/train_cca.py to train the model.```

```Use code/model/category_evaluation.py to evaluate for ECAE task```

```Use code/model/clause_evaluation.py to evaluate for ECE task```

</li>
</ul>

</body>
</html>