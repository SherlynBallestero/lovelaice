from typing import List 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.functional import split


class Chunk:
    def __init__(self, text:str, *, rewrite:str=None) -> None:
        self.text = text
        self.rewrite = rewrite or text


class Document:
    def __init__(self, raw) -> None:
        self.raw = raw
        self.sentences = self._split(self.raw)
        self.chunks = []
        self.model= SentenceTransformer('all-MiniLM-L6-v2') 
    

    def _split(self, text: str):
        return [s.strip() + "." for s in text.split(".") if s]

    def chunk(self, size:int, overlap:int=0) -> list[str]:
        self.chunks = list(self._chunksSimilarityWay())

    def _chunks(self, size, overlap):
        current = []

        for s in self.sentences:
            current.append(s)

            if len(current) == size:
                yield Chunk(" ".join(current))

                if overlap > 0:
                    current = current[-overlap:]
                else:
                    current = []

        if current:
            yield Chunk(" ".join(current))
            
    #cambios a partir de aqui         
    def _chunksSimilarityWay(self):
        marcked=self.coloringSimilarity()
        pivot=[]
        for i in range(len(marcked)):
            current=[]
            if marcked[i] in pivot:
             continue
            else:
                pivot.append(marcked[i])
                for j in range(len(marcked)):
                    if marcked[j]==marcked[i]:
                        current.append(self.sentences[j])
                        #sentence+=str(self.sentences[j])
                    
                yield Chunk(" ".join(current))
        
    #colorear el texto segun similitud
    def coloringSimilarity(self):
        tensores=self.TensorText()
        marck=[-2]*len(tensores)
        for i in range(len(tensores)):
            for j in range(len(tensores)):
                if marck[j]!=-2:
                    continue
                else:
                    p=self.similarityCos(tensores[i],tensores[j])
                    if p>=0.6:
                        marck[j]=i
        return marck               
    #concirtiendo una oracion a tensor
    def ConvertToTensorSentence(self,sentence,model):  
        embedding1 = model.encode(sentence, convert_to_tensor=True)  
        return embedding1 
    #tensor de cada oracion de un texto
    def TensorText(self):   
        answer=list(range(len(self.sentences)))
        for i in range(len(self.sentences)):
            answer[i]=self.ConvertToTensorSentence(self.sentences[i],self.model)
        return answer
    #similitud entre dos tensores
    def similarityCos(self,tensor1,tensor2):
        similarity=cosine_similarity(tensor1.reshape(1,-1),tensor2.reshape(1,-1))[0][0]
        return similarity                    