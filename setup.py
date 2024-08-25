from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Sentimatrix',
    version='0.1.0',
    description='Advanced sentiment analysis platform for text, web content, audio data and Image data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Siddharth Magesh',
    author_email='siddharthmagesh007@gmail.com',
    url='https://github.com/Siddharth-magesh/Sentimatrix', 
    packages=find_packages(),
    install_requires=[
        'transformers',
        'Pillow',
        'torch',
        'groq',
        'openai',
        'deep_translator',
        'matplotlib',
        'seaborn',
        'pandas',
        'SpeechRecognition',
        'requests',
        'beautifulsoup4',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
