import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 然后运行训练
import sys
sys.argv = ['train.py', '--env', 'CartPole-v1', '--episodes', '200']

exec(open('train.py').read())
