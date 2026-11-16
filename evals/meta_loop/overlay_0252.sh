#!/bin/bash
# Usage: overlay_0252.sh NAME   (report 0252) maturin release wheel -> /home/dsweet/bench_work/NAME (src/enn + wheel .so + ennbo.libs)
set -e; B=/home/dsweet/bench_work; R=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf $B/wheel_$1 $B/$1 $B/whx_$1; mkdir -p $B/$1
(cd $R && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2} maturin build --release -o $B/wheel_$1 2>&1 | tail -1)
cp -r $R/src/enn $B/$1/
python -c "import zipfile,glob,sys; zipfile.ZipFile(glob.glob('$B/wheel_$1/*.whl')[0]).extractall('$B/whx_$1')"
cp $B/whx_$1/enn/*.so $B/$1/enn/; cp -r $B/whx_$1/ennbo.libs $B/$1/; rm -rf $B/whx_$1
echo overlay $B/$1
