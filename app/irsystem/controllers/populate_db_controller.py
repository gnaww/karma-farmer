from . import *
from app.irsystem.models.helpers import *
from app.db import populate_db, Data
import traceback


@irsystem.route("/populateDB", methods=["GET"])
def populate():
    try:
        populate_db()
    except Exception as err:
        traceback.print_exc()
        return {"success": False, "error": "{0}".format(err)}

    return {"success": True}
