#!/bin/bash

type=$1
export FLASK_APP=index.py

if [ "$#" -ne 1 ]; then
    echo "Argument error! Use ./deploy.sh [p production |d development]."
    exit 1
fi

if [[ "p" = "$type" ]]; then
	echo "motion2prog explorer being deployed in production mode..."
	python -m flask run --host=0.0.0.0
elif [[ "d" = "$type" ]]; then
	echo "motion2prog explorer being deployed in development mode..."
	export FLASK_ENV=development
	python -m flask run
else 
    echo "Argument error! Use ./deploy.sh [p production |d development]."
fi
