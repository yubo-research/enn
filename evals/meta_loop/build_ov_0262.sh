set -e
N=${1:-ov_0262}
rm -rf /home/dsweet/bench_work/wheel_$N /home/dsweet/bench_work/$N /home/dsweet/bench_work/whx_$N
mkdir -p /home/dsweet/bench_work/$N
cd /home/dsweet/projects/enn
CARGO_BUILD_JOBS=2 maturin build --release -o /home/dsweet/bench_work/wheel_$N 2>&1 | tail -1
cp -r src/enn /home/dsweet/bench_work/$N/
python -c "import zipfile,glob; zipfile.ZipFile(glob.glob('/home/dsweet/bench_work/wheel_$N/*.whl')[0]).extractall('/home/dsweet/bench_work/whx_$N')"
cp /home/dsweet/bench_work/whx_$N/enn/*.so /home/dsweet/bench_work/$N/enn/
cp -r /home/dsweet/bench_work/whx_$N/ennbo.libs /home/dsweet/bench_work/$N/ 2>/dev/null || true
rm -rf /home/dsweet/bench_work/whx_$N
echo overlay done $N
