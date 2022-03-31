# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:13:18 2022

@author: Isha
"""

import spacy
nlp = spacy.load('en_core_web_sm')

import neuralcoref

def get_resolved_entities(filename):
    neuralcoref.add_to_pipe(nlp)
    with open(filename, 'r', encoding="UTF-8") as file:
        text = file.read()
        doc=nlp(text)
        print(doc._.coref_resolved)


    # doc = nlp(u'A blockchain is a distributed database that is shared among the nodes of a computer network. As a database, a blockchain stores information electronically in digital format. Blockchains are best known for their crucial role in cryptocurrency systems, such as Bitcoin, for maintaining a secure and decentralized record of transactions. The innovation with a blockchain is that it guarantees the fidelity and security of a record of data and generates trust without the need for a trusted third party. One key difference between a typical database and a blockchain is how the data is structured. A blockchain collects information together in groups, known as blocks, that hold sets of information. Blocks have certain storage capacities and, when filled, are closed and linked to the previously filled block, forming a chain of data known as the blockchain. All new information that follows that freshly added block is compiled into a newly formed block that will then also be added to the chain once filled. A database usually structures its data into tables, whereas a blockchain, like its name implies, structures its data into chunks (blocks) that are strung together. This data structure inherently makes an irreversible time line of data when implemented in a decentralized nature. When a block is filled, it is set in stone and becomes a part of this time line. Each block in the chain is given an exact time stamp when it is added to the chain.')