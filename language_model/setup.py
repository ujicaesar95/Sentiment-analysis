from setuptools import setup

setup(name='languagemodel',
      packages=[
            'spellcorrector',
            'spellcorrector.pickled',
            'spellcorrector.corpus',
      ],
      version='0.1',
      include_package_data=True
      )
