"""
논블로킹 단일키 입력 (터미널 raw mode).

캡처 툴은 ego pose 를 계속 폴링하면서 동시에 키를 받아야 하므로 input() 을 쓸 수 없다.
with KeyReader() as kr: kr.poll() -> 'e' 또는 None.

prompt() 는 raw mode 를 잠시 풀고 일반 input() 으로 문자열을 받는다
(신호등 ID 처럼 여러 글자를 입력받을 때 사용).
"""

import select
import sys
import termios
import tty


class KeyReader(object):
    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self.fd = self.stream.fileno()
        self._saved = None
        self.enabled = self.stream.isatty()

    def __enter__(self):
        if self.enabled:
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        self.restore()
        return False

    def restore(self):
        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def poll(self, timeout=0.0):
        """키가 눌려 있으면 한 글자, 아니면 None."""
        if not self.enabled:
            return None
        r, _, _ = select.select([self.stream], [], [], timeout)
        if not r:
            return None
        return self.stream.read(1)

    def prompt(self, message, default=""):
        """raw mode 를 잠시 풀고 한 줄 입력을 받는다."""
        saved = self._saved
        if saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, saved)
        try:
            sys.stdout.write("\n" + message)
            sys.stdout.flush()
            line = sys.stdin.readline()
            if line is None:
                return default
            line = line.strip()
            return line if line else default
        finally:
            if saved is not None:
                tty.setcbreak(self.fd)

    def confirm(self, message, default=True):
        hint = "[Y/n]" if default else "[y/N]"
        ans = self.prompt("%s %s " % (message, hint), "").lower()
        if not ans:
            return default
        return ans.startswith("y")
