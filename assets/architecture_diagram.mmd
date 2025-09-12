```mermaid
graph TB
    %% Data Sources and Filtering
    A[Google Patents Public Dataset<br/>2.6TB - Global Publications] --> B{Multi-Stage Filtering}
    B --> |publication_date: 2017-01 to 2025-02<br/>LENGTH title_en > 0<br/>LENGTH abstract_en > 0<br/>English Language Publications| C[Filtered Enterprise Dataset<br/>49M+ English Patents]
    
    %% Table Optimization
    C --> D[BigQuery Table Optimization<br/>patents_2017_2025_en]
    D --> E[PARTITION BY publication_date<br/>CLUSTER BY publication_number, country_code<br/>Monthly Partitions for Cost Optimization]
    
    %% Subset Selection for Demo
    E --> F{Demo Subset Selection}
    F --> |Focus: 2024-01 to 2024-06<br/>High Quality Patents: title_en >= 30<br/>AND abstract_en >= 100| G[Working Dataset<br/>2.9M Patents - 6 Months]
    
    %% Embedding Generation Pipeline
    G --> H[Patent Text Extraction<br/>title_en + abstract_en]
    H --> I[Sentence Transformers<br/>all-MiniLM-L6-v2<br/>384-dimensional vectors]
    I --> |56x faster than<br/>ML.GENERATE_EMBEDDING<br/>6.5 hours vs 15 days| J[Vector Embeddings<br/>2.9M × 384 dimensions]
    
    %% BigQuery AI Integration
    J --> K[(BigQuery Embeddings Table<br/>patent_embeddings_local)]
    K --> L[Table Optimization<br/>PARTITION BY publication_date<br/>CLUSTER BY publication_number, country_code]
    L --> M[CREATE VECTOR INDEX<br/>IVF Indexing with STORING]
    
    %% Search Pipeline
    N[User Query<br/>Natural Language] --> O[Query Embedding<br/>Sentence Transformers]
    O --> P[BigQuery AI VECTOR_SEARCH<br/>With Partition Pruning]
    P --> Q[Semantic Similarity<br/>Cosine Distance + Filtering]
    
    %% Optimized Storage and Processing
    M --> P
    E --> |Partition Pruning<br/>80-85% Cost Reduction| P
    Q --> R[Top-K Similar Patents<br/>- Document Similarity Ranking<br/>- Sentence-Level Explainability]
    
    %% User Interfaces - Split into two paths
    R --> S1[Streamlit Cloud Demo<br/>- Interactive Dashboard<br/>- Real-time Search & Explainability<br/>- Performance Metrics<br/>- User-Friendly Interface]
    R --> S2[Kaggle Notebook<br/>- Technical Implementation<br/>- Code Demonstration<br/>- Static Demo<br/>- Performance Analysis]
    
    %% Performance Monitoring
    U[Performance Metrics] --> V[Query Latency: <5s<br/>Partition Efficiency: 80-85% cost reduction<br/>Discoverability: 98%+<br/>Scalability: 49M+ patents]
    P --> U
    
    %% Architecture Benefits
    W[Production Optimizations] --> X[Monthly Partitioning<br/>Smart Clustering<br/>Vector Indexing<br/>Hybrid Architecture<br/>Enterprise Scalability]
    
    %% Data Flow Annotations
    Y[Data Scale Progression] --> Z[2.6TB → 49M patents → 14.4GB → 2.9M subset<br/>Production filtering → Optimized storage → Demo+Metrics focus]
    
    %% Demo Platform Comparison
    AA[Demo Platform Benefits] --> BB[Streamlit: Interactive UI, Local Control<br/>Kaggle: Cloud Performance, GPU Processing<br/>BigQuery Proximity, Production Metrics]
    
    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef filtering fill:#f3e5f5
    classDef optimization fill:#fff3e0
    classDef processing fill:#f3e5f5
    classDef bigqueryAI fill:#fff3e0
    classDef userInterface fill:#e8f5e8
    classDef metrics fill:#fff8e1
    
    class A,C dataSource
    class B,F filtering
    class D,E,K,L,M optimization
    class H,I,J processing
    class O,P,Q bigqueryAI
    class N,S1,S2,T1,T2 userInterface
    class U,V,W,X,Y,Z,AA,BB metrics