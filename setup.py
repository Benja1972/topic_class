from setuptools import setup


with open('README.md',"r") as f:
    long_description = f.read()

setup(name='topic_class',
      version='0.2.0',
      description='This package consists of Topic Model class ',
      url='https://github.com/Benja1972/topic_class',
      author='Sergei Rybalko',
      author_email='benja1972@gmail.com',
      license='MIT',
      packages=['topic_class'],
      install_requires=['gensim', 'numpy', 'joblib', 'scikit-learn'],
      python_requires='>=3.7',
      include_package_data=True,
      zip_safe=True)
