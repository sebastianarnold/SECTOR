# SECTOR: A Neural Model for Coherent Topic Segmentation and Classification

This is a fork of [TeXoo 1.1.2](https://github.com/sebastianarnold/TeXoo) that contains a standalone implementation of SECTOR from the following paper:

Sebastian Arnold, Rudolf Schneider, Philippe Cudré-Mauroux, Felix A. Gers and Alexander Löser. [SECTOR: A Neural Model for Coherent Topic Segmentation and Classification](https://transacl.org/ojs/index.php/tacl/article/view/1540/359)

```
@article{arnold2019sector,
  author = {Arnold, Sebastian and Schneider, Rudolf and Cudré-Mauroux, Philippe and Gers, Felix A. and Löser, Alexander},
  title = {SECTOR: A Neural Model for Coherent Topic Segmentation and Classification},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {7},
  pages = {169-184},
  year = {2019},
  doi = {10.1162/tacl\_a\_00261}
}
```

The corresponding WikiSection Dataset can be obtained [here](https://github.com/sebastianarnold/WikiSection).

## Getting Started

These instructions will get you a copy of SEC up and running on your local machine for development and testing purposes. If you are going to use TeXoo as a Maven dependency only, you might skip this step.

### Prerequisites

The following dependencies are required if you are planning to run SECTOR locally. They are already contained in the Dockerfile:

- **Oracle Java 8 JDK**
- **Apache Maven** Build system for Java  
<https://maven.apache.org/guides/index.html>

### Installation

First we need to build a docker image with all dependencies:

- run ```bin/docker-install```

### Usage

There exist several run scripts in the `bin/` directory. You can start them right in the docker container:

- ***Training:*** run ```bin/docker-run sector-train [args]```

```
usage: sector-train [-e <arg>] [-h] -i <arg> [-l <arg>] -o <arg> [-t
       <arg>] [-u] [-v <arg>]
SECTOR: train SectorAnnotator from WikiSection dataset
 -e,--embedding <arg>    path to word embedding model, will use bloom
                         filters if not given
 -h,--headings           train multi-label model (SEC>H), otherwise
                         single-label model (SEC>T) is used
 -i,--input <arg>        file name of WikiSection training dataset
 -l,--language <arg>     language to use for sentence splitting and
                         stopwords (EN or DE)
 -o,--output <arg>       path to create and store the model
 -t,--test <arg>         file name of WikiSection test dataset (will test
                         after training if given)
 -u,--ui                 enable training UI (http://127.0.0.1:9000)
 -v,--validation <arg>   file name of WikiSection validation dataset (will
                         use early stopping if given)
```

- ***Evaluation:*** run ```bin/docker-run sector-eval [args]```
```
usage: sector-eval [-e <arg>] -m <arg> -t <arg>
SECTOR: evaluate SectorAnnotator on WikiSection dataset
 -e,--embedding <arg>   search path to word embedding models (if not
                        provided by the model itself)
 -m,--model <arg>       path to the pre-trained model
 -t,--test <arg>        file name of WikiSection test dataset
```

## License

   Copyright 2019 Sebastian Arnold, Alexander Löser, Rudolf Schneider

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
