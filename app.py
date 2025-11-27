# app.py
import os
import re
import tempfile

import jieba
import pdfplumber
from docx import Document
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from cluster_kmeans import Embedder, KMeansCluster, resource_path

app = FastAPI()

# 初始化模型（在实际应用中，您可能需要调整路径）
MODEL_PATH = resource_path('bge-small-zh-v1.5')


class DocumentReader:
    """统一的文档读取器，支持PDF、DOCX、DOC和TXT文件"""
    with open(resource_path('static/stopwords.txt'), 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]

    @staticmethod
    def get_paragraphs(text):
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef。！？.!?]', '', text)

        sentences = []
        for sentence in re.split(r'[。！？.!?]', text):
            if sentence.strip():
                text = [word for word in jieba.cut(sentence) if word not in DocumentReader.stopwords and len(word) > 1]
                text = ''.join(text).strip()
                sentences.append(text)

        # 可以选择将短句子合并
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= 200:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "。"

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def read_pdf(file_path):
        """读取PDF文件，按句子分割"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'

            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return []

    @staticmethod
    def read_docx(file_path):
        """读取DOCX文件，按句子分割"""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = '\n'.join(full_text)

            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return []

    @staticmethod
    def read_txt(file_path):
        """读取TXT文件，按句子分割"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            return text
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return []


@app.post("/classify-document/")
async def classify_document(file: UploadFile = File(...)):
    """
    接收文档文件并返回主题分类结果
    支持PDF、DOCX、DOC和TXT文件
    """
    # 检查文件类型
    allowed_types = ["application/pdf",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are allowed")

    try:
        # 创建临时文件来保存上传的文档
        file_extension = ""
        if file.content_type == "application/pdf":
            file_extension = ".pdf"
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_extension = ".docx"
        elif file.content_type == "text/plain":
            file_extension = ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # 根据文件类型读取内容
        document_reader = DocumentReader()
        text = ""
        if file.content_type == "application/pdf":
            text = document_reader.read_pdf(tmp_file_path)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = document_reader.read_docx(tmp_file_path)
        elif file.content_type == "text/plain":
            text = document_reader.read_txt(tmp_file_path)

        # 检查是否有内容
        if not text.strip():
            raise HTTPException(status_code=400, detail="No content found in document")

        # 进行嵌入
        embedder = Embedder(MODEL_PATH)
        paragraphs = document_reader.get_paragraphs(text)
        chunk_embeddings = embedder.embed(paragraphs)

        # 进行聚类
        cluster_labels, kmeans = KMeansCluster.perform_kmeans_clustering(chunk_embeddings)
        cluster_names = KMeansCluster.extract_semantic_cluster_names(
            chunk_embeddings, cluster_labels, paragraphs, kmeans
        )

        # 清理临时文件
        os.unlink(tmp_file_path)

        # 选择样本数最多的聚类（而不是盲目使用标签 0）
        try:
            import numpy as _np

            unique, counts = _np.unique(cluster_labels, return_counts=True)
            largest_label = int(unique[_np.argmax(counts)])
        except Exception:
            largest_label = 0

        # 获取聚类名称，若为空则回退到 label 0 或通用占位名称
        cluster_name = cluster_names.get(largest_label) or cluster_names.get(0) or f"聚类_{largest_label}"
        # 返回关键词列表
        return JSONResponse(content=cluster_name.split(' '))

    except Exception as e:
        # 确保临时文件被清理
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
