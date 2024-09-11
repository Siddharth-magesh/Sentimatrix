from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Sentimatrix',
    version='0.1.4',
    description='Advanced sentiment analysis platform for text, web content, audio data and Image data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Siddharth Magesh',
    author_email='siddharthmagesh007@gmail.com',
    url='https://github.com/Siddharth-magesh/Sentimatrix',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'transformers',
        'Pillow',
        'torch==2.3.1',
        'groq',
        'openai',
        'deep_translator',
        'matplotlib',
        'seaborn',
        'pandas',
        'SpeechRecognition',
        'requests',
        'beautifulsoup4',
        'accelerate',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
