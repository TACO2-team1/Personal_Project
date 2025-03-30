import matplotlib.pyplot as plt
import numpy
import os

#17epoch
def draw_pnet_train():
    EPOCHES=20
    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1)
    plt.title("PNet classification loss (BCELoss)")

    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/pnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[6]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/pnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        print(filename)
        print(strs)
        test_counter.append(i*412564)
        test_losses.append(float(strs[3]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('binary cross entropy loss')

    plt.subplot(2,2,2)
    plt.title("PNet boundingbox loss (MSELoss)")
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/pnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[8]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/pnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[5]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('mean square error loss')

    plt.subplot(2,2,3)
    plt.title("PNet landmark loss (MSELoss)")
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/pnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[10]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/pnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[7]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('mean square error loss')


    plt.subplot(2,2,4)
    plt.title("PNet loss sum")
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/pnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[12]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/pnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[9]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('loss sum')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=1.1, top=0.9, wspace=0.25, hspace=0.5)
    

    plt.savefig("imgs/PnetTrain.png")
    plt.show()



#45epoch
def draw_rnet_train():
    EPOCHES=50
    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1)
    plt.title("RNet classification loss (BCELoss)")

    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/rnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[6]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/rnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        print(filename)
        print(strs)
        test_counter.append(i*412564)
        test_losses.append(float(strs[3]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('binary cross entropy loss')

    plt.subplot(2,2,2)
    plt.title("RNet boundingbox loss (MSELoss)")
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/rnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[8]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/rnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[5]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('mean square error loss')

    plt.subplot(2,2,3)
    plt.title("RNet landmark loss (MSELoss)")
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/rnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[10]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/rnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[7]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('mean square error loss')


    plt.subplot(2,2,4)
    plt.title("RNet loss sum")
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/rnet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[12]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/rnet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[9]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('loss sum')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=1.1, top=0.9, wspace=0.25, hspace=0.5)
    

    plt.savefig("imgs/RnetTrain.eps")
    plt.show()


#epoch44
def draw_onet_train():
    EPOCHES=50
    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1)
    plt.title("ONet classification loss (BCELoss)")
    # plt.ylim(0,0.4)

    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/onet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[6]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/onet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        print(filename)
        print(strs)
        test_counter.append(i*412564)
        test_losses.append(float(strs[3]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('binary cross entropy loss')

    plt.subplot(2,2,2)
    plt.title("ONet boundingbox loss (MSELoss)")
    # plt.ylim(0,0.04)
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/onet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[8]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/onet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[5]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('mean square error loss')

    plt.subplot(2,2,3)
    plt.title("ONet landmark loss (MSELoss)")
    # plt.ylim(0,0.05)
    
    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/onet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[10]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/onet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[7]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('mean square error loss')


    plt.subplot(2,2,4)
    plt.title("ONet loss sum")
    # plt.ylim(0,0.4)

    train_counter=[]
    train_losses=[]

    for i in range(EPOCHES):
        filename='logs/onet/epoch{}.txt'.format(i+1)
        file=open(filename)
        for j in range(65):
            if j%5==0:
                strs=file.readline().split()
                train_counter.append(412564*i+j*6400)
                train_losses.append(float(strs[12]))
    
    test_counter=[]
    test_losses=[]
    for i in range(EPOCHES+1):
        filename='logs/onet/test_after_epoch{}.txt'.format(i)
        file=open(filename)
        strs=file.readline().split()
        test_counter.append(i*412564)
        test_losses.append(float(strs[9]))
    
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('loss sum')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=1.1, top=0.9, wspace=0.25, hspace=0.5)
    

    plt.savefig("imgs/OnetTrain.eps")
    plt.show()


def draw_pgd_step_size():
    step_sizes = [0.0,0.25,0.5,0.75,
                  1.0,1.25,1.5,1.75,
                  2.0,2.25,2.5,2.75,
                  3.0,3.25,3.5,3.75,
                  4.0,4.25,4.5,4.75,
                  5.0,5.25,5.5,5.75,6.0]
    logs_path = 'logs/_step_size_pgd'

    xs = []
    ys = []

    for step_size in step_sizes:
        path = os.path.join(logs_path,'_step_size_for_pgd{}.txt'.format(step_size))
        file = open(path)
        strs = file.readline().split()
        print(strs)
        xs.append(step_size)
        ys.append(float(strs[2]))

    x1s = xs.copy()
    y1s = [0.053]*len(x1s)

    
    plt.figure(figsize=(10,7))
    plt.title("Probablity of misdetection")
    plt.plot(xs, ys, color='blue')
    plt.plot(x1s, y1s, color='red')
    plt.legend(['PGD', 'clean'], loc='upper right')
    plt.xlabel(r'$\log_2\frac{Step\ Size\ Choosen\ in\ PGD}{0.001}$')
    plt.ylabel('Probablity')
    plt.savefig('imgs/_pgd.eps')
    plt.show()

def draw_pgd_eps():
    xs=[0.01,0.02,0.03,0.04,0.05]
    ys=[0.106,0.331,0.545,0.588,0.890]
    y1s=[0.053]*5

    plt.figure(figsize=(10,7))
    plt.title("Probablity of misdetection")
    plt.plot(xs, ys, color='blue')
    plt.plot(xs, y1s, color='red')
    plt.legend(['PGD', 'clean'], loc='upper right')
    plt.xlabel('Epsilon')
    plt.ylabel('Probablity')
    plt.savefig('imgs/_pgd_eps.eps')
    plt.show()

if __name__=='__main__':
    #17
    # draw_pnet_train()
    #45
    # draw_rnet_train()
    #44
    # draw_onet_train()
    # draw_pgd_step_size()
    draw_pgd_eps()