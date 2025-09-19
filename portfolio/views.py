from django.shortcuts import render


def home_view(request):
    """Main portfolio landing page"""
    context = {
        'name': 'Dali Ben Jemaa',
        'title': 'Machine Learning Engineer',
        'email': 'dali@dalibenj.com',
        'github': 'https://github.com/dalibenj',
        'linkedin': 'https://linkedin.com/in/dalibenj',
        'skills': [
            'Python', 'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'Django', 'FastAPI', 'TensorFlow', 'PyTorch', 'Scikit-learn',
            'Docker', 'AWS', 'PostgreSQL', 'Redis', 'Git'
        ],
        'projects': [
            {
                'title': 'RAG-Powered Q&A System',
                'description': 'Built a Retrieval-Augmented Generation system using LangChain, FAISS, and DeepSeek API for intelligent document search and question answering.',
                'tech_stack': ['Python', 'Django', 'LangChain', 'FAISS', 'DeepSeek API', 'Docker'],
                'github_url': 'https://github.com/dalibenj/daliBenJDotCom',
                'demo_url': '/rag/'
            },
            {
                'title': 'Computer Vision Model',
                'description': 'Developed a custom CNN model for image classification with 95% accuracy on custom dataset.',
                'tech_stack': ['Python', 'TensorFlow', 'OpenCV', 'NumPy', 'Matplotlib'],
                'github_url': '#',
                'demo_url': '#'
            },
            {
                'title': 'NLP Text Analysis Tool',
                'description': 'Created a sentiment analysis and text summarization tool using transformer models.',
                'tech_stack': ['Python', 'Transformers', 'Hugging Face', 'Streamlit', 'NLTK'],
                'github_url': '#',
                'demo_url': '#'
            }
        ],
        'experience': [
            {
                'company': 'Tech Company',
                'position': 'Senior ML Engineer',
                'duration': '2022 - Present',
                'description': 'Led development of ML pipelines and deployed models serving 1M+ users.'
            },
            {
                'company': 'Startup Inc',
                'position': 'ML Engineer',
                'duration': '2020 - 2022',
                'description': 'Built recommendation systems and NLP models for content analysis.'
            }
        ],
        'education': [
            {
                'degree': 'Master of Science in Computer Science',
                'school': 'University Name',
                'year': '2020',
                'focus': 'Machine Learning & AI'
            }
        ]
    }
    return render(request, 'portfolio/home.html', context)
