from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = '7.DocumentLoaders\Social_Network_Ads.csv')

data = loader.load()

print(len(data))
print(data[0])