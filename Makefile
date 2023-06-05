# Syntax: BRANCH=<Branch Name> make delete_branch
delete_branch:
	git checkout master
	git branch -d $(BRANCH)


# Syntax: BRANCH=<Branch Name> make create_new_branch
create_new_branch:
	git checkout -b $(BRANCH)
	git pull origin master
