pip install -r requirements.txt

# change pandas version restriction in opencompass
git clone -b 0.4.2 https://github.com/open-compass/opencompass.git
sed -i.bak "s/pandas<2.0.0/pandas/" ./opencompass/requirements/runtime.txt
pip install git+https://github.com/ShaohonChen/PyExt.git@a95f488490fc57ec17d0c00a99c6bc0a4726824f ./opencompass
rm -rf ./opencompass

clone and install LiveCodeBench and human-eval as git submodules

cd ./benchmark/LiveCodeBench
bash install.sh
cd ../human-eval
bash install.sh
