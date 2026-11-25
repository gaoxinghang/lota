"""定义临时要求和常见敏感信息的过滤规则。"""

import re

TEMPORARY = re.compile(
  r"本次|这次|当次|这一轮|本轮|仅此一次|当前任务|当前会话|本会话|临时|暂时|"
  r"不要记住|不要保存|别记住|"
  r"\b(?:this (?:session|turn|task|time)|for now|today only|"
  r"just this once|do not (?:remember|save))\b", re.IGNORECASE,
)
SENSITIVE = re.compile(
  r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk-|ghp_|github_pat_)\S+|"
  r"\bBearer\s+\S+|"
  r"(?:password|passwd|api[ _-]?key|secret|token|密码|口令|密钥|令牌)"
  r"\s*(?:[:：=]|是)\s*\S+", re.IGNORECASE,
)
