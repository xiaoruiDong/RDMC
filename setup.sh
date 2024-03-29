# Install rdmc
pip install -e $(dirname $0) --no-deps -v

# Give xtb_gaussian.pl executable permission
chmod +x $(dirname $0)/rdmc/external/inpwriter/xtb_gaussian.pl