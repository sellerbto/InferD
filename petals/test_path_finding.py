


class Vertex:
    def __init__(self, id, stage):
        self.stage = stage
        self.id = id

    def get_weight(self, other):
        return 
    
    def __repr__(self):
        return f'{self.id}_{self.stage}'


n = 10
ids = list(range(n))

vertices = [Vertex(id, stage) for id, stage in zip(ids, [i//2 for i in range(n)])]
print(vertices)