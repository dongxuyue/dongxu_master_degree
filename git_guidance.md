# Git&Github

**本文介绍git及github基本概念，以及git常用命令，和管理github仓库的方法。**

## Git与Github基本概念

​		git是一个版本管理程序，由C语言编写，是一种分布式管理工具。也就是说每个参与到git管理项目中的人都会有一份工程的完整拷贝，可以用在局域网中，也可以用于互联网。Git支持版本回退，可以随时随地回退到以前的工程中，避免丢失。

**总结：**

- 方便协作
- 分布式保障
- 可以回退版本，避免丢失

​		github是一个开源代码仓库，相当于一个git的节点，但GitHub的优势在于他是7*24小时运行的，因此可以随时随地与它来同步你的工程。

## Git常用命令

### 初始化&提交

- 设置git的用户名和邮件地址，global参数表示本台机器上所有仓库均使用这个配置

```
$ git config --global user.name "Your Name"
$ git config --global user.email "email@example.com"
```

- 创建repository，即代码仓库，相当于一个被git管理的文件夹

```
$ git init
Initialized empty Git repository in /Users/michael/learngit/.git/
```

此时文件夹中会多出来一个隐藏文件，不要手欠去改它

- 添加本次要提交的文件，可以添加多个

```
$ git add yourfile
```
- 提交刚才添加的一个或多个文件，-m参数表示本次提交的说明，最好写上
```
$ git commit -m "wrote a readme file"
```

​		注意一定要把要提交的文件放在repository内，而且保证已经创建了repository

### 版本状态查看

- 查看当前的文件与repository中文件的差别

```
$ git status
```

​		这行命令会告诉我们哪些东西被修改了

- 查看修改到底改了什么

```
$ git diff
```

​		可以在命令后边加上文件名，否则会显示所有文件的修改

- 查看文件内容

```
$ cat filename
```



### 版本回退

- 查看历史版本信息，会按照时间线列出

```
$ git log
```

- 回退版本

```
$ git reset --hard HEAD^
```

​		HEAD表示当前的版本，一个^代表回退一个版本

- 版本复原

  有时候，你回退了又后悔了怎么办？

  两种方式：

  - 使用```git log```的时候记下来commit id
  
  - 使用命令
  
    ```
    $ git reflog
    ```
  
    可以查看全部的commit id
  
  ```
  $ git reset --hard 1094a
  ```

ps：HEAD是一个指针，以上的操作都是在移动这个指针。

### 撤销修改

- 当修改了文件内容，还没有加入到暂存区时

  ```
  $ git checkout -- filename
  ```

  可以把文件内容恢复到repository中的状态

- 当已经加入了暂存区，撤销后回到加入暂存区的状态

- 撤销已经加入暂存区的修改

  ```
  $ git reset HEAD readme.txt
  ```

  此时暂存区会变干净，如果想把当前工作区文件再恢复到之前的状态，可以再使用checkout

### 删除文件

- 从repository中删除一个文件

```
$ git rm test.txt
```

- 如果本地删除了，想要恢复

```
$ git checkout -- test.txt
```

由此可以看出，checkout只是用repository库中的文件替换本地文件

### 远程连接（github）

**远程连接首先需要创建本地的密钥，包含公钥和私钥**

```
$ ssh-keygen -t rsa -C "youremail@example.com"
```

这里的邮箱应该填写GitHub的注册邮箱，执行完此条命令，去主目录找到.ssh文件夹，把公钥信息粘贴到github的仓库设置页面。

- 查看是否连接成功

  ```
  $ ssh -T git@github.com
  ```

  如若让选择就无脑yes

- 本地需要执行的命令

  ```
  $ git remote add origin git@github.com:NICHOLASFATHER/dongxu_master_degree.git
  ```

  origin为远程仓库的默认名字，也可以叫别的

- 第一次把本地仓库的内容送到GitHub上

  ```
  $ git push -u origin master
  ```

  当提交过一次后，之后的命令可以简化，不需要加-u参数

  ```
  $ git push origin master
  ```

- 删除链接

  ```
  $ git remote rm origin
  ```

  仅解绑，不删除任何文件

- 克隆GitHub仓库到本地

  ```
  $ git clone git@github.com:michaelliao/gitskills.git
  ```

  可以克隆自己的仓库，也可以克隆别人的

### 分支管理

分支是一种非常有效安全的技术手段，每当进行一个work的时候，最好都新建一个branch，做完工作后再合并到master上。

- 创建分支

  ```
  $ git switch -c dev
  ```

  switch用于转换分支，-c表示创建

- 查看当前分支

  ```
  $ git branch
  * dev
    master
  ```

  在当前分支完成提交后可以切换回主分支

- 切换回主分支

  ```
  $ git switch master
  ```

- 在主分支下合并dev到主分支

  ```
  $ git merge dev
  ```

  若此时仅修改一个分支再合并，不会产生合并冲突

- 删除分支

  ```
  $ git branch -d dev
  ```

- 分支合并冲突

  当遇到冲突时（即两个分支都修改了某个文件时），git不能帮我们自动合并，而是要进行手动操作，操作完成后记得commit，删除子分支。

- **多人协作**

  - 简单项目，单分支

    可以直接在master上进行提交

    ```
    git push origin master
    ```

  - 复杂项目，多分支

    往往使用一个master分支用于稳定版发布，一个dev分支用于大家开发

    所有的开发都要先提交到dev分支上，**必须在本地创建同名分支才可以提交**！

    ```
    git switch -c dev
    git add sth
    git commit -m "sth"
    git push origin dev
    ```

  因此，多人协作的工作模式通常是这样：

  1. 首先，可以试图用`git push origin `推送自己的修改；
  2. 如果推送失败，则因为远程分支比你的本地更新，需要先用`git pull`试图合并；
  3. 如果合并有冲突，则解决冲突，并在本地提交；
  4. 没有冲突或者解决掉冲突后，再用`git push origin `推送就能成功！

  如果`git pull`提示`no tracking information`，则说明本地分支和远程分支的链接关系没有创建，用命令`git branch --set-upstream-to  origin/`。

## Reference

1. https://www.liaoxuefeng.com/wiki/896043488029600
2. https://www.runoob.com/w3cnote/git-guide.html