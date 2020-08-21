#!/usr/bin/env bash
# Script to download a Wikipedia dump

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="data"
DUMP_DIR="${ROOT}/wiki_dumps"
EXTR_DIR="${ROOT}/wiki_extr"
WIKI_DIR="${ROOT}/wiki"
EXTR="wikiextractor"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"
mkdir -p "${EXTR_DIR}"
mkdir -p "${WIKI_DIR}"

echo "Saving data in ""$ROOT"
LANG="ca"
echo "Chosen language: ""$LANG"
DUMP_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_PATH="${DUMP_DIR}/${DUMP_FILE}"

if [ ! -f "${DUMP_PATH}" ]; then
   echo "Starting download..."
   wget -c "https://dumps.wikimedia.org/""${LANG}""wiki/latest/""${DUMP_FILE}""" -P "${DUMP_DIR}"
else
  echo "${DUMP_PATH} already exists. Skipping download."
fi

if [ ! -d "${EXTR}" ]; then
 git clone https://github.com/attardi/wikiextractor.git
 cd "${EXTR}"
 git checkout 9cf2a2a883fc8e2146ff8df234d036e695df1be4
 cd ..
fi


EXTR_PATH="${EXTR_DIR}/${LANG}"
if [ ! -d "${EXTR_PATH}" ]; then
  echo "Extracting ${DUMP_PATH} to ${EXTR_PATH}..."
  python wikiextractor/WikiExtractor.py -s --json -q -o "${EXTR_PATH}" "${DUMP_PATH}"
else
  echo "${EXTR_PATH} already exists. Skipping extraction."
fi
