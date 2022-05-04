import requests
import pickle 
import time
from tqdm import tqdm


ISBN = pickle.load(open("ISBN.pkl", "rb"))

bookData = dict()


def split(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

ISBN_batches = list(split(ISBN, 49_000))



batch_iter = 5


for i, isbn in enumerate(tqdm(ISBN_batches[batch_iter])):
    r=requests.get(f"https://api.pro.isbndb.com/book/{isbn}", 
    headers={"Authorization":"47797_a7abaf827d0519e361ba24c0ab2638c4"})

    bookData[isbn] = r.json()

    time.sleep(0.25)

    if i % 10 == 0 and i > 0:
        pickle.dump(bookData, open(f"summaryBatch/batch_{batch_iter}.pkl", "wb"))
        #pickle.dump(bookData, open(f"summaryBatch/batch_{batch_iter}_part2.pkl", "wb"))

#pickle.dump(bookData, open(f"summaryBatch/batch_{batch_iter}_part2.pkl", "wb"))
pickle.dump(bookData, open(f"summaryBatch/batch_{batch_iter}.pkl", "wb"))
