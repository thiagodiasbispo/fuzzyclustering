## Lista de conteúdos

* [Informações gerais](#Informações-gerais)
* [Configuração do ambiente](#configuracao-do-ambiente)
* [Instalação das dependências via Anaconda](#instalacao-via-anaconda)
* [Iniciando o projeto](#iniciando-projeto)
* [Configurando o ambiente do zero](#configurando-do-zero)
* [Exportando ambiente](#exportando-ambiente)

## Informações gerais
Projeto criado como parte da primeira nota da disciplina de Aprendizagem de Máquina da UFPE 2020.1
	
## Configuração do ambiente
Este projeto tem as seguintes dependências:
* Python: 3.6
* Pandas
* Numpy
* Scikit-learn
* Jupyter
* autopep8
	
## Instalação das dependências via Anaconda
* Passo 1 - Download e instalação do [Anaconda](https://www.anaconda.com/products/individual#Downloads)
* Passo 2 - Download do projeto: 
	* ```$ git clone https://github.com/thiagodiasbispo/fuzzyclustering.git ```
* Passo 3 - Instalar dependências (Linux) - Com o prompt aberto na pasta do projeto: 
	* ```$ conda create -n fuzzy --file conda_requirements_linux.txt ```

## Iniciando o projeto
* Passo 1 - Iniciar o ambiente virtual do Anaconda:
	* No linux: ```$ conda activate fuzzy ```
* Passo 2 - Executar o seguinte comando na pasta do projeto: 
	* ```$ jupyter notebook```

## Configurando o ambiente do zero
* Passo 1 - Download e instalação do [Anaconda](
ttps://www.anaconda.com/products/individual#Downloads)
* Passo 2 - Criando o ambiente virtual:
	* ```$ conda create -n fuzzy python=3.6 ```
* Passo 3 - Instalando as dependências do projeto:
	* ```$ conda install pandas numpy jupyter autopep8```
	* ```$ conda install -c anaconda scikit-learn ```
	* ```$ conda install -c conda-forge jupyter_contrib_nbextensions ``` 

## Exportando ambiente:

* Linux - Com o abmiente virtual ativado, executar na pasta do projeto: 
	* ```$ conda list -e > conda_requirements_linux.txt ```