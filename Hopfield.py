import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import time

class Hopfield:
    w=[]
    theta = []
    def __init__(self,matrixs):
        print("start training")
        matrixs = np.array(matrixs)
        n = len(matrixs)
        self.j_row = len(matrixs[0])
        self.j_column = len(matrixs[0][0])
        self.p = self.j_row*self.j_column
        print(f"array shape = {self.j_row},{self.j_column}")

        print(f"len = {self.p}")
        self.w = np.zeros((self.p, self.p))
        for matrix in matrixs:
            print(np.array(matrix))
            matrix =np.array([matrix.flatten()]).T
            #print("flatten = \n",matrix)
            #print(f"transpose : \n{matrix.T}")
            self.w += np.dot(matrix ,matrix.T)
        
        self.w = self.w/self.p - n/self.p*np.eye(self.p)
        print(self.w)
        self.theta = np.array([np.sum(self.w, axis=1)]).T
        #print("theta:",self.theta)
        print("traning complete")
        
        
    def predict_sync(self,matrix,ax = None,canvas = None,status_panel = None):
        print("start testing(sync).")
        last_result = [-1]
        result = []
        matrix = self.array_flatten(matrix)
        while not np.array_equal(last_result,result):
            self.recall_status_panel_update(status_panel)  
            last_result = result.copy()
            result = []
            
            result = np.dot(self.w,matrix)
            # print("result dot:\n",result)
            #result = result - self.theta
            # print("result -theta:\n",result)
            
    
            for j,row in enumerate(result):
                result[j][0] =self.sign(row[0],matrix[j][0])
        
            #print("result sign:\n",result)
            self.refresh_picture(result.reshape(self.j_row,self.j_column),ax,canvas)
            matrix = result.copy()
        
        self.recall_status_panel_update(status_panel,done=True)               
        print("testing complete(sync).")
        return result.reshape(self.j_row,self.j_column)
    
    def predict_async(self,matrix,max_epoch = 10,ax = None,canvas = None,status_panel = None):
        print("start testing(async).")
        matrix = self.array_flatten(matrix)
        last_result = []
        #print(matrix)
        for epoch in range(max_epoch):
            self.recall_status_panel_update(status_panel,epoch)
            if np.array_equal(matrix,last_result):
                max_epoch = epoch
                break
            last_result = matrix.copy()
            print(f"epoch : {epoch}")
            for j in range(self.p):
                y = 0
                # for i in range(self.p):
                #     y += self.w[j][i] * matrix[i][0]
                y = np.dot(self.w[j],matrix)
                #y -= self.theta[j][0]
                matrix[j][0] = self.sign(y,matrix[j][0])
                self.refresh_picture(matrix.reshape(self.j_row,self.j_column),ax,canvas)
            
        print("testing complete(async).")
        self.recall_status_panel_update(status_panel,max_epoch,True)
        self.refresh_picture(matrix.reshape(self.j_row,self.j_column),ax,canvas)

        return matrix.reshape(self.j_row,self.j_column)
    
    
    
    def recall_status_panel_update(self,status_panel,epoch = None,done = False):
        if status_panel:
            if not done:
                status_panel.set(f"\n回想中...\n")
            if done:
                status_panel.set(f"\n回想完成\n")
                
            if epoch!=None:
                status_panel.set(status_panel.get()+f"Epoch = {epoch}\n")
    
    
    def sign(self,value,last_value):
        if value > 0:
            return 1
        elif value == 0:
            return last_value
        else:
            return -1
        
    def array_flatten(self,matrix):
        matrix = np.array(matrix)
        matrix = np.array([matrix.flatten()])
        matrix = matrix.T
        return matrix
    
    def refresh_picture(self,result,ax,canvas):
        if (ax!=None) and (canvas!=None):
            #print("refresh!")
            ax.cla()
            ax.imshow(result, cmap='gray', interpolation='nearest')
            canvas.draw()
            canvas.flush_events()
            #time.sleep(0.05)
            
    

        
    
        
        
#測試輸入    
if __name__ == "__main__":
    
    matrix = [[[1],[-1],[1]],[[-1],[1],[-1]]]
    
    test = Hopfield(matrix)
    result = test.predict_async([[1],[1],[-1]])
    print(f"final result :\n {result}")