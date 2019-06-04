function user = getCurrentUser()
% function returns user name of PC

if isunix()
    user = getenv('USER');
else
    user = getenv('username');
end