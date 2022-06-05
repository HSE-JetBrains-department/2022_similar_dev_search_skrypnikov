#!usr/bin/sh

if ! [ -d "build" ]; then
    mkdir build
fi

pushd build

if [ -d "go-enry" ]; then
    rm -rf go-enry
fi

git clone git@github.com:go-enry/go-enry.git
pushd go-enry && make static && popd
pushd go-enry/python

if ! [ -z "$(command -v pyenv)" ]; then
    echo 'USING PYENV'
    pyenv exec pip install -r requirements.txt
    pyenv exec python setup.py bdist_wheel --universal
    pyenv exec pip install "$(find ./dist -name enry-*.whl)" --force-reinstall
else
    echo 'NOT USING PYENV'
    pip install -r requirements.txt
    python setup.py bdist_wheel --universal
    pip install dist/enry-*.whl --force-reinstall
fi

popd && popd