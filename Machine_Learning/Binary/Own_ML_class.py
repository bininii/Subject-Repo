import torch.nn as nn

# class 가져다가 불러서 쓸 수는 있어야 됨!
class BinaryClassification(nn.Module):
    def __init__(self, in_dim, out_dim):   # 생성자를 통해 객체 생성
        super().__init__() # super: 부모에 있는 모든것들을 가져다가 씀
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    """
    model = BinaryClassification(2, 1)
    hx = model(x) : forward method로 수행됨.
    
    """
    def forward(self, x):   # forward 라는 명칭으로만 사용되어야 함. => forward: method
        return self.sigmoid(self.linear(x))
