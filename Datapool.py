import random
from torch.utils.data import Dataset, DataLoader
class Datapool():
    def __init__(self, warmstart_percentage, train_dl):
        # self.train_dl = train_dl #
        # print("total_num:",len(train_dl.dataset))
        self.train_data = self.dataloader_to_list(train_dl)
        self.total_num = len(self.train_data)
        # print("total_num:",self.total_num)
        annotated_num = int(self.total_num * warmstart_percentage)
        annotated_indices = random.sample(range(self.total_num), annotated_num)
        annotated_indices.sort()
        self.annotated_data = [self.train_data[i] for i in annotated_indices]
        self.unannotated_data = [self.train_data[i] for i in range(self.total_num) if i not in annotated_indices]
        # print("len(annotated_data)",len(self.annotated_data))
        # print("len(unannotated_data)",len(self.unannotated_data))
        # print("self.annotated_data[0]",self.annotated_data[0])

        self.an_dataloader = DataLoader(self.annotated_data, batch_size=8, shuffle=False, drop_last=True, num_workers=2)
        self.un_dataloader = DataLoader(self.unannotated_data, batch_size=8, shuffle=False, drop_last=True, num_workers=2)

    def dataloader_to_list(self, dataloader):
        data_list = []
        cnt = 0
        for batch in dataloader:  
            for i in range(dataloader.batch_size):      
                data_list.append({
                    'source_ids':batch['source_ids'][i],
                    'target_ids':batch['target_ids'][i],
                    'source_mask':batch['source_mask'][i],
                    'target_mask':batch['target_mask'][i]
                })
        return data_list

    def getAnnotatedData(self):
        return self.annotated_data
    def getAnnotatedDataloader(self):
        return self.an_dataloader
    def getUnannotatedData(self):
        return self.unannotated_data
    def getUnannotatedDataloader(self):
        return self.un_dataloader
    def addAnnotatedData(self, selected_indices=[]):
        annotated_indices = selected_indices
        # annotated_indices.sort()
        self.annotated_data = self.annotated_data + [self.unannotated_data[i] for i in annotated_indices]
        self.unannotated_data = [self.unannotated_data[i] for i in range(len(self.unannotated_data)) if i not in annotated_indices]
        self.an_dataloader = DataLoader(self.annotated_data, batch_size=8, shuffle=True, drop_last=True, num_workers=2)
        self.un_dataloader = DataLoader(self.unannotated_data, batch_size=8, shuffle=False, drop_last=True, num_workers=2)
        self.showDetail()
    def showDetail(self):
        print("annotated_samples_num:", len(self.an_dataloader.dataset))
        print("unannotated_samples_num:", len(self.un_dataloader.dataset))
