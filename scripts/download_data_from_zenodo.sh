cd "$(dirname "$0")"/..
wget https://zenodo.org/api/records/17113741/files/data.tar.gz/content
tar -xvzf content
rm content