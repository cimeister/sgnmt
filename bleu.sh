#!/bin/bash
N=7283
filename=$1
ref=$2
for arg in "$@"; do
    case $1 in  
    	-n)  shift  
			N=$1;;
		-a|--all) 
			N=$(wc -l $filename | awk '{ print $1 }');;
    esac
    shift
    case $1 in  
    	-r)  shift
		baseline=$1
		head -n $N $ref > "tmp"
		head -n $N $baseline | perl scripts/detokenizer.perl -l en > "out"
		cat out | sacrebleu --width 2 tmp;;
		
    esac
    shift
done

head -n $N $ref > "tmp"
head -n $N $filename | perl scripts/detokenizer.perl -l en > "out"

cat out | sacrebleu --width 4 tmp